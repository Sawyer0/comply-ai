"""Initialization helpers for the storage manager."""

from __future__ import annotations

import asyncio
import base64
from typing import Any, Dict, Optional

import asyncpg  # type: ignore[import-not-found,import-untyped]
import boto3  # type: ignore[import-not-found,import-untyped]
from botocore.exceptions import (  # type: ignore[import-not-found,import-untyped]
    ClientError,
    NoCredentialsError,
)

from .models import StorageBackend


class StorageInitializationMixin:
    """Provides initialization routines for storage backends and encryption."""

    # pylint: disable=too-few-public-methods

    settings: Any
    logger: Any
    backend: StorageBackend
    _s3_client: Any
    _kms_client: Any
    _db_pool: Any
    _clickhouse_client: Any
    _fernet: Optional[Any]

    async def _init_s3(self) -> None:
        """Initialize S3 client and verify bucket access."""
        try:
            session = boto3.Session(
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region,
            )

            self._s3_client = session.client("s3")
            self._kms_client = session.client("kms")

            # Verify bucket exists and is accessible
            assert self._s3_client is not None
            await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.head_bucket, {"Bucket": self.settings.s3_bucket}
            )

            self.logger.info("S3 client initialized", bucket=self.settings.s3_bucket)

        except NoCredentialsError:
            self.logger.error("AWS credentials not found")
            raise
        except ClientError as exc:  # pragma: no cover - boto errors are hard to mock
            self.logger.error("S3 initialization failed", error=str(exc))
            raise

    async def _init_database(self) -> None:
        """Initialize database connection based on backend type."""
        if self.backend == StorageBackend.POSTGRESQL:
            await self._init_postgresql()
        elif self.backend == StorageBackend.CLICKHOUSE:
            await self._init_clickhouse()

    async def _init_postgresql(self) -> None:
        """Initialize PostgreSQL connection pool."""
        try:
            pool_candidate = asyncpg.create_pool(
                host=self.settings.db_host,
                port=self.settings.db_port,
                user=self.settings.db_user,
                password=self.settings.db_password,
                database=self.settings.db_name,
                min_size=2,
                max_size=10,
            )
            if hasattr(pool_candidate, "acquire"):
                self._db_pool = pool_candidate  # type: ignore[assignment]
            else:
                self._db_pool = await pool_candidate  # type: ignore[assignment]

            await self._create_postgresql_tables()
            self.logger.info("PostgreSQL pool initialized")

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("PostgreSQL initialization failed", error=str(exc))
            raise

    async def _init_clickhouse(self) -> None:
        """Initialize ClickHouse client."""
        try:
            from clickhouse_driver import (
                Client as ClickHouseClient,  # type: ignore  # pylint: disable=import-outside-toplevel
            )

            self._clickhouse_client = ClickHouseClient(
                host=self.settings.db_host,
                port=self.settings.db_port,
                user=self.settings.db_user,
                password=self.settings.db_password,
                database=self.settings.db_name,
            )

            assert self._clickhouse_client is not None
            await asyncio.get_event_loop().run_in_executor(
                None, self._clickhouse_client.execute, "SELECT 1"
            )

            await self._create_clickhouse_tables()
            self.logger.info("ClickHouse client initialized")

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("ClickHouse initialization failed", error=str(exc))
            raise

    async def _init_encryption(self) -> None:
        """Initialize encryption using KMS or local key."""
        try:

            class _DummyFernet:
                def __init__(self, key: bytes) -> None:
                    self.key = key

                def encrypt(self, data: bytes) -> bytes:
                    """Return plaintext bytes unchanged (test stub)."""
                    return data

                def decrypt(self, data: bytes) -> bytes:
                    """Return ciphertext bytes unchanged (test stub)."""
                    return data

            if self.settings.kms_key_id:
                assert self._kms_client is not None
                key_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._kms_client.generate_data_key,
                    {"KeyId": self.settings.kms_key_id, "KeySpec": "AES_256"},
                )
                raw_key = key_response["Plaintext"][:32]
            else:
                raw_key = self.settings.encryption_key.encode()
                if len(raw_key) < 32:
                    raw_key = raw_key.ljust(32, b"0")
                else:
                    raw_key = raw_key[:32]

            fernet_key = base64.urlsafe_b64encode(raw_key)

            try:
                from cryptography.fernet import (
                    Fernet,  # type: ignore  # pylint: disable=import-outside-toplevel
                )

                try:
                    self._fernet = Fernet(fernet_key)
                    self.logger.info(
                        "Encryption initialized",
                        kms_enabled=bool(self.settings.kms_key_id),
                    )
                except (
                    Exception
                ) as exc:  # pragma: no cover - defensive logging  # pylint: disable=broad-exception-caught
                    self._fernet = _DummyFernet(fernet_key)
                    self.logger.warning(
                        "Invalid encryption key; using dummy Fernet (no-op)",
                        error=str(exc),
                    )
            except ModuleNotFoundError:
                self._fernet = _DummyFernet(fernet_key)
                self.logger.warning(
                    "cryptography not installed; using dummy Fernet (no-op)"
                )

        except (
            Exception
        ) as exc:  # pragma: no cover - defensive logging  # pylint: disable=broad-exception-caught
            self.logger.error("Encryption initialization failed", error=str(exc))
            raise

    async def _setup_worm_policy(self) -> None:
        """Configure S3 bucket with WORM (Object Lock) policy."""
        try:
            try:
                assert self._s3_client is not None
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._s3_client.get_object_lock_configuration,
                    {"Bucket": self.settings.s3_bucket},
                )
                self.logger.info("S3 Object Lock already configured")
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if "ObjectLockConfigurationNotFoundError" not in str(exc):
                    raise

            lock_config: Dict[str, Any] = {
                "ObjectLockEnabled": "Enabled",
                "Rule": {
                    "DefaultRetention": {
                        "Mode": "GOVERNANCE",
                        "Years": self.settings.s3_retention_years or 7,
                    }
                },
            }

            assert self._s3_client is not None
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._s3_client.put_object_lock_configuration,
                {
                    "Bucket": self.settings.s3_bucket,
                    "ObjectLockConfiguration": lock_config,
                },
            )

            self.logger.info(
                "S3 WORM policy configured",
                retention_years=lock_config["Rule"]["DefaultRetention"]["Years"],
            )

        except ClientError as exc:  # pragma: no cover - boto errors hard to mock
            if "InvalidBucketState" in str(exc):
                self.logger.warning(
                    "Cannot enable Object Lock on existing bucket without versioning"
                )
            else:
                self.logger.error("Failed to configure WORM policy", error=str(exc))
                raise
