"""S3 persistence helpers for the storage manager."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from ..tenant_isolation import TenantContext
from .models import StorageRecord


class StorageS3Mixin:
    """Utilities for persisting and reading records from S3."""

    # pylint: disable=too-few-public-methods

    settings: Any
    logger: Any
    tenant_manager: Any
    _s3_client: Any

    async def _store_in_s3(self, record: StorageRecord) -> str:
        """Store record in S3 with immutable configuration."""
        s3_key = f"records/{record.timestamp.strftime('%Y/%m/%d')}/{record.id}.json"

        record_data = {
            "id": record.id,
            "source_data": record.source_data,
            "mapped_data": record.mapped_data,
            "model_version": record.model_version,
            "timestamp": record.timestamp.isoformat(),
            "metadata": record.metadata,
            "encrypted": record.encrypted,
        }

        assert self._s3_client is not None
        params = {
            "Bucket": self.settings.s3_bucket,
            "Key": s3_key,
            "Body": json.dumps(record_data),
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": self.settings.kms_key_id or None,
            "ObjectLockMode": "GOVERNANCE",
            "ObjectLockRetainUntilDate": datetime.now(timezone.utc)
            + timedelta(days=365 * (self.settings.s3_retention_years or 7)),
        }
        params = {key: value for key, value in params.items() if value is not None}

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._s3_client.put_object(**params),
        )

        return s3_key

    async def _retrieve_from_s3(
        self, record_id: str, tenant_context: Optional[TenantContext]
    ) -> Optional[StorageRecord]:
        """Retrieve record from S3 by searching for the record ID."""
        try:
            assert self._s3_client is not None
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self._s3_client.list_objects_v2,
                {
                    "Bucket": self.settings.s3_bucket,
                    "Prefix": "records/",
                    "MaxKeys": 1000,
                },
            )

            for obj in response.get("Contents", []):
                if record_id in obj["Key"]:
                    assert self._s3_client is not None
                    obj_response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self._s3_client.get_object,
                        {"Bucket": self.settings.s3_bucket, "Key": obj["Key"]},
                    )

                    record_data = json.loads(obj_response["Body"].read())

                    record = StorageRecord(
                        id=record_data["id"],
                        source_data=record_data["source_data"],
                        mapped_data=record_data["mapped_data"],
                        model_version=record_data["model_version"],
                        timestamp=datetime.fromisoformat(record_data["timestamp"]),
                        metadata=record_data["metadata"],
                        tenant_id=record_data.get("tenant_id", "unknown"),
                        s3_key=obj["Key"],
                        encrypted=record_data.get("encrypted", False),
                    )

                    if (
                        tenant_context is None
                        or self.tenant_manager.validate_tenant_access(
                            tenant_context.tenant_id, record.tenant_id, "read"
                        )
                    ):
                        return record

        except (
            Exception
        ) as exc:  # pragma: no cover - defensive logging  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to retrieve from S3", record_id=record_id, error=str(exc)
            )

        return None


__all__ = ["StorageS3Mixin"]
