"""Persistent detector registry repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from shared.database.connection_manager import get_service_db


@dataclass
class DetectorRecord:
    id: str
    detector_type: str
    detector_name: str
    endpoint_url: str
    health_check_url: Optional[str]
    status: str
    version: str
    capabilities: List[str]
    configuration: Dict[str, Any]
    tenant_id: str
    created_at: str
    updated_at: str
    last_health_check: Optional[str]
    health_status: Optional[str]
    response_time_ms: Optional[int]
    error_rate: Optional[float]


class DetectorRepository:
    """Repository providing CRUD operations for detectors table."""

    def __init__(self, *, service_name: str = "orchestration") -> None:
        self._db = get_service_db(service_name)

    async def list_detectors(
        self,
        *,
        tenant_id: Optional[str] = None,
        detector_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[DetectorRecord]:
        conditions: List[str] = []
        params: List[Any] = []

        if tenant_id:
            params.append(tenant_id)
            conditions.append(f"tenant_id = ${len(params)}")
        if detector_type:
            params.append(detector_type)
            conditions.append(f"detector_type = ${len(params)}")
        if status:
            params.append(status)
            conditions.append(f"status = ${len(params)}")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = (
            "SELECT * FROM detectors "
            f"{where_clause} ORDER BY tenant_id, detector_type, detector_name"
        )
        rows = await self._db.fetch(query, *params)
        return [self._row_to_record(row) for row in rows]

    async def get_detector(self, detector_id: str) -> Optional[DetectorRecord]:
        row = await self._db.fetchrow("SELECT * FROM detectors WHERE id = $1", detector_id)
        return self._row_to_record(row) if row else None

    async def detector_exists(
        self,
        *,
        tenant_id: str,
        detector_name: str,
        detector_type: str,
    ) -> bool:
        record = await self.get_detector_by_identity(
            tenant_id=tenant_id,
            detector_name=detector_name,
            detector_type=detector_type,
        )
        return record is not None

    async def get_detector_by_identity(
        self,
        *,
        tenant_id: str,
        detector_name: str,
        detector_type: str,
    ) -> Optional[DetectorRecord]:
        query = (
            "SELECT * FROM detectors WHERE tenant_id = $1 AND detector_name = $2 AND detector_type = $3"
        )
        row = await self._db.fetchrow(query, tenant_id, detector_name, detector_type)
        return self._row_to_record(row) if row else None

    async def get_detector_by_name(
        self,
        *,
        tenant_id: str,
        detector_name: str,
    ) -> Optional[DetectorRecord]:
        query = "SELECT * FROM detectors WHERE tenant_id = $1 AND detector_name = $2"
        row = await self._db.fetchrow(query, tenant_id, detector_name)
        return self._row_to_record(row) if row else None

    async def create_detector(self, *, fields: Mapping[str, Any]) -> str:
        insert_sql = (
            """
            INSERT INTO detectors (
                id, detector_type, detector_name, endpoint_url, health_check_url,
                status, version, capabilities, configuration, tenant_id, last_health_check,
                health_status, response_time_ms, error_rate
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10, $11,
                $12, $13, $14
            )
            RETURNING id
            """
        )
        new_id = await self._db.fetchval(
            insert_sql,
            fields.get("id") or str(uuid4()),
            fields["detector_type"],
            fields["detector_name"],
            fields["endpoint_url"],
            fields.get("health_check_url"),
            fields.get("status", "active"),
            fields["version"],
            fields.get("capabilities", []),
            fields.get("configuration", {}),
            fields["tenant_id"],
            fields.get("last_health_check"),
            fields.get("health_status", "unknown"),
            fields.get("response_time_ms"),
            fields.get("error_rate", 0.0),
        )
        return new_id

    async def update_detector(self, detector_id: str, *, fields: Dict[str, Any]) -> bool:
        if not fields:
            return False
        assignments = ", ".join(
            f"{column} = ${index}"
            for index, column in enumerate(fields.keys(), start=2)
        )
        query = f"UPDATE detectors SET {assignments}, updated_at = NOW() WHERE id = $1"
        result = await self._db.execute(query, detector_id, *fields.values())
        return result[-1] != "0"

    async def delete_detector(self, detector_id: str) -> bool:
        result = await self._db.execute("DELETE FROM detectors WHERE id = $1", detector_id)
        return result[-1] != "0"

    async def update_health_by_name(
        self,
        detector_name: str,
        *,
        health_status: str,
        response_time_ms: Optional[int],
        error_rate: Optional[float] = None,
    ) -> None:
        query = (
            """
            UPDATE detectors
            SET health_status = $1,
                last_health_check = NOW(),
                response_time_ms = $2,
                error_rate = COALESCE($3, error_rate)
            WHERE detector_name = $4
            """
        )
        await self._db.execute(
            query,
            health_status,
            response_time_ms,
            error_rate,
            detector_name,
        )

    @staticmethod
    def _row_to_record(row: Any) -> DetectorRecord:
        return DetectorRecord(
            id=str(row["id"]),
            detector_type=row["detector_type"],
            detector_name=row["detector_name"],
            endpoint_url=row["endpoint_url"],
            health_check_url=row["health_check_url"],
            status=row["status"],
            version=row["version"],
            capabilities=list(row["capabilities"] or []),
            configuration=dict(row["configuration"] or {}),
            tenant_id=row["tenant_id"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_health_check=str(row["last_health_check"]) if row["last_health_check"] else None,
            health_status=row["health_status"],
            response_time_ms=row["response_time_ms"],
            error_rate=float(row["error_rate"]) if row["error_rate"] is not None else None,
        )
