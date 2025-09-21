"""Database-specific helpers for the storage manager."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Optional

from .models import StorageBackend, StorageRecord
from ..tenant_isolation import TenantContext


POSTGRES_INSERT_SQL = """
    INSERT INTO storage_records (
        id,
        source_data,
        mapped_data,
        model_version,
        timestamp,
        metadata,
        tenant_id,
        encrypted
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
"""


class StorageDatabaseMixin:
    """Helpers for persisting and reading records from database backends."""

    # pylint: disable=too-few-public-methods

    backend: StorageBackend
    logger: Any
    tenant_manager: Any
    settings: Any
    _db_pool: Any
    _clickhouse_client: Any

    async def _store_in_database(self, record: StorageRecord) -> None:
        """Store record in the configured database backend."""
        if self.backend == StorageBackend.POSTGRESQL:
            await self._store_postgresql(record)
        else:
            await self._store_clickhouse(record)

    async def _store_postgresql(self, record: StorageRecord) -> None:
        """Store record in PostgreSQL."""
        assert self._db_pool is not None
        pool = self._db_pool
        ctx = pool.acquire()
        if hasattr(ctx, "__aenter__"):
            async with ctx as conn:  # type: ignore[func-returns-value]
                await conn.execute(
                    POSTGRES_INSERT_SQL,
                    record.id,
                    record.source_data,
                    record.mapped_data,
                    record.model_version,
                    record.timestamp,
                    record.metadata,
                    record.tenant_id,
                    record.encrypted,
                )
        else:
            awaited = await ctx  # type: ignore[misc]
            if hasattr(awaited, "__aenter__"):
                async with awaited as conn:  # type: ignore[func-returns-value]
                    await conn.execute(
                        POSTGRES_INSERT_SQL,
                        record.id,
                        record.source_data,
                        record.mapped_data,
                        record.model_version,
                        record.timestamp,
                        record.metadata,
                        record.tenant_id,
                        record.encrypted,
                    )
            else:
                conn = awaited
                try:
                    await conn.execute(
                        POSTGRES_INSERT_SQL,
                        record.id,
                        record.source_data,
                        record.mapped_data,
                        record.model_version,
                        record.timestamp,
                        record.metadata,
                        record.tenant_id,
                        record.encrypted,
                    )
                finally:
                    if hasattr(pool, "release"):
                        rel = pool.release(conn)
                        if asyncio.iscoroutine(rel):
                            await rel

    async def _store_clickhouse(self, record: StorageRecord) -> None:
        """Store record in ClickHouse."""
        assert self._clickhouse_client is not None
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._clickhouse_client.execute,
            """
            INSERT INTO storage_records
            (id, source_data, mapped_data, model_version, timestamp, metadata, tenant_id, encrypted)
            VALUES
            """,
            [
                (
                    record.id,
                    record.source_data,
                    record.mapped_data,
                    record.model_version,
                    record.timestamp,
                    str(record.metadata),
                    record.tenant_id,
                    record.encrypted,
                )
            ],
        )

    async def _retrieve_from_database(
        self, record_id: str, tenant_context: TenantContext
    ) -> Optional[StorageRecord]:
        """Retrieve record from database."""
        if self.backend == StorageBackend.POSTGRESQL:
            return await self._retrieve_postgresql(record_id, tenant_context)
        return await self._retrieve_clickhouse(record_id, tenant_context)

    async def _retrieve_postgresql(
        self, record_id: str, tenant_context: TenantContext
    ) -> Optional[StorageRecord]:
        """Retrieve record from PostgreSQL with tenant filtering."""
        base_query = """
            SELECT id, source_data, mapped_data, model_version, timestamp, metadata, tenant_id, s3_key, encrypted
            FROM storage_records WHERE id = $1
        """

        filtered_query = self.tenant_manager.apply_tenant_filter(
            base_query, tenant_context
        )

        assert self._db_pool is not None
        pool = self._db_pool
        ctx = pool.acquire()
        if hasattr(ctx, "__aenter__"):
            async with ctx as conn:  # type: ignore[func-returns-value]
                row = await conn.fetchrow(filtered_query, record_id)
        else:
            awaited = await ctx  # type: ignore[misc]
            if hasattr(awaited, "__aenter__"):
                async with awaited as conn:  # type: ignore[func-returns-value]
                    row = await conn.fetchrow(filtered_query, record_id)
            else:
                conn = awaited
                try:
                    row = await conn.fetchrow(filtered_query, record_id)
                finally:
                    if hasattr(pool, "release"):
                        rel = pool.release(conn)
                        if asyncio.iscoroutine(rel):
                            await rel

        if row:
            return StorageRecord(
                id=row.get("id"),  # type: ignore[index]
                source_data=row.get("source_data"),
                mapped_data=row.get("mapped_data"),
                model_version=row.get("model_version"),
                timestamp=row.get("timestamp"),
                metadata=row.get("metadata", {}),
                tenant_id=row.get("tenant_id", "unknown"),
                s3_key=row.get("s3_key"),
                encrypted=row.get("encrypted", False),
            )
        return None

    async def _retrieve_clickhouse(
        self, record_id: str, tenant_context: TenantContext
    ) -> Optional[StorageRecord]:
        """Retrieve record from ClickHouse with tenant filtering."""
        base_query = "SELECT * FROM storage_records WHERE id = %s"
        filtered_query = self.tenant_manager.apply_tenant_filter(
            base_query, tenant_context
        )

        assert self._clickhouse_client is not None
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._clickhouse_client.execute, filtered_query, [record_id]
        )

        if result:
            row = result[0]
            metadata = json.loads(row[5]) if isinstance(row[5], str) else row[5]
            return StorageRecord(
                id=row[0],
                source_data=row[1],
                mapped_data=row[2],
                model_version=row[3],
                timestamp=row[4],
                metadata=metadata,
                tenant_id=row[6],
                s3_key=row[7] if len(row) > 7 else None,
                encrypted=row[8] if len(row) > 8 else False,
            )
        return None

    async def _update_s3_key(self, record_id: str, s3_key: str) -> None:
        """Update the S3 key in the database record."""
        if self.backend != StorageBackend.POSTGRESQL:
            return

        assert self._db_pool is not None
        pool = self._db_pool
        ctx = pool.acquire()
        if hasattr(ctx, "__aenter__"):
            async with ctx as conn:  # type: ignore[func-returns-value]
                await conn.execute(
                    "UPDATE storage_records SET s3_key = $1 WHERE id = $2",
                    s3_key,
                    record_id,
                )
        else:
            awaited = await ctx  # type: ignore[misc]
            if hasattr(awaited, "__aenter__"):
                async with awaited as conn:  # type: ignore[func-returns-value]
                    await conn.execute(
                        "UPDATE storage_records SET s3_key = $1 WHERE id = $2",
                        s3_key,
                        record_id,
                    )
            else:
                conn = awaited
                try:
                    await conn.execute(
                        "UPDATE storage_records SET s3_key = $1 WHERE id = $2",
                        s3_key,
                        record_id,
                    )
                finally:
                    if hasattr(pool, "release"):
                        rel = pool.release(conn)
                        if asyncio.iscoroutine(rel):
                            await rel

    async def _create_postgresql_tables(self) -> None:
        """Create PostgreSQL tables if they don't exist."""
        assert self._db_pool is not None
        pool = self._db_pool
        ctx = pool.acquire()
        if hasattr(ctx, "__aenter__"):
            async with ctx as conn:  # type: ignore[func-returns-value]
                await conn.execute(self._postgres_schema_sql)
        else:
            awaited = await ctx  # type: ignore[misc]
            if hasattr(awaited, "__aenter__"):
                async with awaited as conn:  # type: ignore[func-returns-value]
                    await conn.execute(self._postgres_schema_sql)
            else:
                conn = awaited
                try:
                    await conn.execute(self._postgres_schema_sql)
                finally:
                    if hasattr(pool, "release"):
                        rel = pool.release(conn)
                        if asyncio.iscoroutine(rel):
                            await rel

    async def _create_clickhouse_tables(self) -> None:
        """Create ClickHouse tables if they don't exist."""
        assert self._clickhouse_client is not None
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._clickhouse_client.execute,
            self._clickhouse_schema_sql,
        )

    async def _cleanup_postgresql(self, cutoff_date: datetime) -> int:
        """Clean up expired PostgreSQL records."""
        assert self._db_pool is not None
        pool = self._db_pool
        ctx = pool.acquire()
        if hasattr(ctx, "__aenter__"):
            async with ctx as conn:  # type: ignore[func-returns-value]
                result = await conn.execute(
                    "DELETE FROM storage_records WHERE timestamp < $1", cutoff_date
                )
        else:
            awaited = await ctx  # type: ignore[misc]
            if hasattr(awaited, "__aenter__"):
                async with awaited as conn:  # type: ignore[func-returns-value]
                    result = await conn.execute(
                        "DELETE FROM storage_records WHERE timestamp < $1", cutoff_date
                    )
            else:
                conn = awaited
                try:
                    result = await conn.execute(
                        "DELETE FROM storage_records WHERE timestamp < $1", cutoff_date
                    )
                finally:
                    if hasattr(pool, "release"):
                        rel = pool.release(conn)
                        if asyncio.iscoroutine(rel):
                            await rel
        try:
            return int(result)  # type: ignore[arg-type]
        except (TypeError, ValueError, AttributeError):
            return int(str(result).split()[-1])

    async def _cleanup_clickhouse(self, _cutoff_date: datetime) -> int:
        """Clean up expired ClickHouse records."""
        assert self._clickhouse_client is not None
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._clickhouse_client.execute,
            "OPTIMIZE TABLE storage_records FINAL",
        )
        return 0

    @property
    def _postgres_schema_sql(self) -> str:
        """Schema used to bootstrap PostgreSQL."""
        return """
                CREATE TABLE IF NOT EXISTS storage_records (
                    id VARCHAR(255) PRIMARY KEY,
                    source_data TEXT NOT NULL,
                    mapped_data TEXT NOT NULL,
                    model_version VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metadata JSONB,
                    tenant_id VARCHAR(100) NOT NULL,
                    s3_key VARCHAR(500),
                    encrypted BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_storage_records_timestamp
                ON storage_records(timestamp);

                CREATE INDEX IF NOT EXISTS idx_storage_records_model_version
                ON storage_records(model_version);

                CREATE INDEX IF NOT EXISTS idx_storage_records_tenant_id
                ON storage_records(tenant_id);

                ALTER TABLE storage_records ENABLE ROW LEVEL SECURITY;

                CREATE POLICY IF NOT EXISTS tenant_isolation_policy ON storage_records
                    FOR ALL
                    USING (tenant_id = current_setting('app.current_tenant_id', true));
            """

    @property
    def _clickhouse_schema_sql(self) -> str:
        """Schema used to bootstrap ClickHouse."""
        return """
            CREATE TABLE IF NOT EXISTS storage_records (
                id String,
                source_data String,
                mapped_data String,
                model_version String,
                timestamp DateTime,
                metadata String,
                tenant_id String,
                s3_key Nullable(String),
                encrypted UInt8,
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (tenant_id, timestamp, id)
            TTL timestamp + INTERVAL 90 DAY
            """


__all__ = ["StorageDatabaseMixin"]
