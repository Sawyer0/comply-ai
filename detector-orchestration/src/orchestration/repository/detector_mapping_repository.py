from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from shared.database.connection_manager import get_service_db


@dataclass
class DetectorMappingConfigRecord:
    id: str
    tenant_id: str
    detector_type: str
    detector_version: Optional[str]
    version: str
    schema_version: str
    mapping_rules: Dict[str, Any]
    validation_schema: Optional[Dict[str, Any]]
    status: str
    is_active: bool
    backward_compatible: bool
    rollback_of_version: Optional[str]
    created_at: str
    activated_at: Optional[str]
    deactivated_at: Optional[str]
    created_by: Optional[str]


class DetectorMappingConfigRepository:
    def __init__(self, *, service_name: str = "orchestration") -> None:
        self._db = get_service_db(service_name)

    async def create_config(self, *, fields: Mapping[str, Any]) -> str:
        insert_sql = (
            """
            INSERT INTO detector_mapping_configs (
                id,
                tenant_id,
                detector_type,
                detector_version,
                version,
                schema_version,
                mapping_rules,
                validation_schema,
                status,
                is_active,
                backward_compatible,
                rollback_of_version,
                created_by
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
                $10,
                $11,
                $12,
                $13
            )
            RETURNING id
            """
        )
        new_id = await self._db.fetchval(
            insert_sql,
            fields.get("id") or str(uuid4()),
            fields["tenant_id"],
            fields["detector_type"],
            fields.get("detector_version"),
            fields["version"],
            fields["schema_version"],
            fields["mapping_rules"],
            fields.get("validation_schema"),
            fields.get("status", "active"),
            fields.get("is_active", False),
            fields.get("backward_compatible", True),
            fields.get("rollback_of_version"),
            fields.get("created_by"),
        )
        return new_id

    async def get_active_config(
        self,
        *,
        tenant_id: str,
        detector_type: str,
    ) -> Optional[DetectorMappingConfigRecord]:
        query = (
            """
            SELECT *
            FROM detector_mapping_configs
            WHERE tenant_id = $1
              AND detector_type = $2
              AND is_active = TRUE
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        row = await self._db.fetchrow(query, tenant_id, detector_type)
        return self._row_to_record(row) if row else None

    async def list_configs(
        self,
        *,
        tenant_id: str,
        detector_type: str,
    ) -> List[DetectorMappingConfigRecord]:
        query = (
            """
            SELECT *
            FROM detector_mapping_configs
            WHERE tenant_id = $1
              AND detector_type = $2
            ORDER BY created_at DESC
            """
        )
        rows = await self._db.fetch(query, tenant_id, detector_type)
        return [self._row_to_record(row) for row in rows]

    async def activate_version(
        self,
        *,
        tenant_id: str,
        detector_type: str,
        version: str,
    ) -> bool:
        # Deactivate currently active configuration(s)
        await self._db.execute(
            """
            UPDATE detector_mapping_configs
            SET is_active = FALSE,
                deactivated_at = NOW()
            WHERE tenant_id = $1
              AND detector_type = $2
              AND is_active = TRUE
            """,
            tenant_id,
            detector_type,
        )

        # Activate requested version
        result = await self._db.execute(
            """
            UPDATE detector_mapping_configs
            SET is_active = TRUE,
                status = 'active',
                activated_at = NOW()
            WHERE tenant_id = $1
              AND detector_type = $2
              AND version = $3
            """,
            tenant_id,
            detector_type,
            version,
        )
        return result[-1] != "0"

    async def rollback_to_version(
        self,
        *,
        tenant_id: str,
        detector_type: str,
        version: str,
    ) -> bool:
        current = await self.get_active_config(
            tenant_id=tenant_id,
            detector_type=detector_type,
        )

        # Mark current active as rolled back
        if current is not None:
            await self._db.execute(
                """
                UPDATE detector_mapping_configs
                SET is_active = FALSE,
                    status = 'rolled_back',
                    deactivated_at = NOW()
                WHERE id = $1
                """,
                current.id,
            )

        # Activate target version and record which version we rolled back from
        result = await self._db.execute(
            """
            UPDATE detector_mapping_configs
            SET is_active = TRUE,
                status = 'active',
                activated_at = NOW(),
                rollback_of_version = $4
            WHERE tenant_id = $1
              AND detector_type = $2
              AND version = $3
            """,
            tenant_id,
            detector_type,
            version,
            current.version if current is not None else None,
        )
        return result[-1] != "0"

    @staticmethod
    def _row_to_record(row: Any) -> DetectorMappingConfigRecord:
        return DetectorMappingConfigRecord(
            id=str(row["id"]),
            tenant_id=row["tenant_id"],
            detector_type=row["detector_type"],
            detector_version=row["detector_version"],
            version=row["version"],
            schema_version=row["schema_version"],
            mapping_rules=dict(row["mapping_rules"] or {}),
            validation_schema=dict(row["validation_schema"])
            if row["validation_schema"] is not None
            else None,
            status=row["status"],
            is_active=bool(row["is_active"]),
            backward_compatible=bool(row["backward_compatible"]),
            rollback_of_version=row["rollback_of_version"],
            created_at=str(row["created_at"]),
            activated_at=str(row["activated_at"])
            if row["activated_at"] is not None
            else None,
            deactivated_at=str(row["deactivated_at"])
            if row["deactivated_at"] is not None
            else None,
            created_by=row["created_by"],
        )
