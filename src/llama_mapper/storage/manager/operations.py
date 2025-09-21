"""High-level operations exposed by the storage manager."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Optional

from ..tenant_isolation import TenantContext
from .models import StorageBackend, StorageRecord, StorageAccessError


class StorageOperationsMixin:
    """Exposes CRUD-style methods built on top of backend mixins."""

    backend: StorageBackend
    logger: Any
    tenant_manager: Any
    privacy_logger: Any
    _fernet: Any

    async def store_record(
        self, record: StorageRecord, tenant_context: Optional[TenantContext] = None
    ) -> str:
        """Store a record in both hot database and cold S3 storage."""
        try:
            if tenant_context and not self.tenant_manager.validate_tenant_access(
                tenant_context.tenant_id, record.tenant_id, "write"
            ):
                raise StorageAccessError(
                    f"Tenant {tenant_context.tenant_id} cannot write to tenant {record.tenant_id}"
                )

            scoped_id = self.tenant_manager.create_tenant_scoped_record_id(
                record.tenant_id, record.id
            )
            original_id = record.id

            db_record = replace(record, id=scoped_id)

            if self._fernet:
                encrypted_source = self._fernet.encrypt(
                    db_record.source_data.encode()
                ).decode()
                encrypted_mapped = self._fernet.encrypt(
                    db_record.mapped_data.encode()
                ).decode()
                db_record.source_data = encrypted_source
                db_record.mapped_data = encrypted_mapped
                db_record.encrypted = True

            await self._store_in_database(db_record)

            s3_key = await self._store_in_s3(db_record)
            db_record.s3_key = s3_key

            await self._update_s3_key(db_record.id, s3_key)

            await self.privacy_logger.log_mapping_success(
                tenant_id=db_record.tenant_id or "unknown",
                detector_type=db_record.metadata.get("detector", "unknown"),
                taxonomy_hit=db_record.metadata.get("taxonomy_hit", "unknown"),
                confidence_score=db_record.metadata.get("confidence", 0.0),
                model_version=db_record.model_version,
                metadata={"s3_key": s3_key, "encrypted": db_record.encrypted},
            )

            self.logger.info(
                "Record stored successfully",
                record_id=db_record.id,
                s3_key=s3_key,
                tenant_id=db_record.tenant_id,
            )
            return original_id

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to store record", record_id=record.id, error=str(exc)
            )
            raise

    async def retrieve_record(
        self, record_id: str, tenant_context: Optional[TenantContext] = None
    ) -> Optional[StorageRecord]:
        """Retrieve a record by ID, checking hot storage first, then S3."""
        try:
            if ":" in record_id:
                scoped_id = record_id
            elif tenant_context is not None:
                scoped_id = self.tenant_manager.create_tenant_scoped_record_id(
                    tenant_context.tenant_id, record_id
                )
            else:
                scoped_id = record_id

            record = await self._retrieve_from_database(
                scoped_id, tenant_context or TenantContext(tenant_id="unknown")
            )

            if not record:
                record = await self._retrieve_from_s3(scoped_id, tenant_context)

            if (
                tenant_context
                and record
                and not self.tenant_manager.validate_tenant_access(
                    tenant_context.tenant_id, record.tenant_id, "read"
                )
            ):
                self.logger.warning(
                    "Tenant access denied for record",
                    requesting_tenant=tenant_context.tenant_id,
                    record_tenant=record.tenant_id,
                    record_id=record_id,
                )
                return None

            if record and record.encrypted and self._fernet:
                record.source_data = self._fernet.decrypt(
                    record.source_data.encode()
                ).decode()
                record.mapped_data = self._fernet.decrypt(
                    record.mapped_data.encode()
                ).decode()

            if record and ":" in record.id:
                _, original_id = self.tenant_manager.extract_tenant_from_record_id(
                    record.id
                )
                record.id = original_id

            return record

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to retrieve record", record_id=record_id, error=str(exc)
            )
            raise

    def _extract_taxonomy_from_mapped_data(self, mapped_data: str) -> Optional[str]:
        """Extract taxonomy label from mapped data JSON."""
        try:
            data = json.loads(mapped_data)
            taxonomy = data.get("taxonomy", [])
            return taxonomy[0] if taxonomy else None
        except (json.JSONDecodeError, IndexError, KeyError):
            return None

    def _extract_confidence_from_mapped_data(self, mapped_data: str) -> Optional[float]:
        """Extract confidence score from mapped data JSON."""
        try:
            data = json.loads(mapped_data)
            value = data.get("confidence")
            if isinstance(value, (int, float)):
                return float(value)
            return None
        except (json.JSONDecodeError, KeyError):
            return None


__all__ = ["StorageOperationsMixin"]
