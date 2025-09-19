"""
Tests for BYOK verification (S3 SSE-KMS) and PII redaction utilities.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_mapper.config.settings import Settings, StorageConfig
from llama_mapper.security.redaction import redact_dict, redact_text
from llama_mapper.storage.manager import StorageManager, StorageRecord


@pytest.mark.asyncio
async def test_s3_put_uses_kms_when_kms_key_id_set():
    storage_cfg = StorageConfig(
        s3_bucket="test-bucket",
        aws_region="us-east-1",
        storage_backend="postgresql",
        db_host="localhost",
        db_port=5432,
        db_name="test",
        db_user="user",
        db_password="pass",
        encryption_key="local-key-32-bytes-string-123456",
        kms_key_id="arn:aws:kms:us-east-1:123456789012:key/test-key",
    )
    settings = Settings(storage=storage_cfg)

    with patch("boto3.Session") as mock_session, patch(
        "asyncpg.create_pool"
    ) as mock_pool, patch.object(StorageManager, "_create_postgresql_tables"), patch.object(
        StorageManager, "_init_encryption"
    ) as mock_init_enc:
        # Mock KMS client for _init_encryption (not executed due to patch)
        mock_s3_client = MagicMock()
        mock_s3_client.head_bucket.return_value = None
        mock_s3_client.put_object.return_value = None
        mock_session.return_value.client.return_value = mock_s3_client

        mock_pool.return_value = AsyncMock()

        mgr = StorageManager(settings.storage)
        await mgr.initialize()

        rec = StorageRecord(
            id="r1",
            source_data="src",
            mapped_data='{"taxonomy": ["PII.Contact.Email"], "confidence": 0.9}',
            model_version="v1",
            metadata={"detector": "d", "tenant_id": "t"},
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )

        await mgr.store_record(rec)

        args, kwargs = mock_s3_client.put_object.call_args
        assert kwargs.get("ServerSideEncryption") == "aws:kms"
        assert kwargs.get("SSEKMSKeyId") == storage_cfg.kms_key_id


def test_redaction_masks_common_pii():
    text = "Contact me at alice@example.com or +1-555-123-4567, SSN 123-45-6789"
    red = redact_text(text)
    assert "[EMAIL_REDACTED]" in red
    assert "[PHONE_REDACTED]" in red
    assert "[SSN_REDACTED]" in red

    data = {
        "prompt": "User said: my card 4111 1111 1111 1111",
        "meta": {"ip": "192.168.1.1", "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.x.y"},
    }
    red_dict = redact_dict(data)
    assert red_dict["prompt"] == "[REDACTED]"
    assert red_dict["meta"]["ip"] == "[IP_REDACTED]"
