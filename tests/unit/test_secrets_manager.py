"""
Unit tests for SecretsManager backends (AWS, Vault, Env).
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from llama_mapper.config.settings import Settings
from llama_mapper.security.secrets_manager import SecretsManager


def test_env_secrets_backend_reads_env():
    os.environ["MY_API_KEY"] = "abc123"
    sm = SecretsManager(Settings(security__secrets_backend="env"))
    assert sm.get("MY_API_KEY") == "abc123"


def test_aws_secrets_backend_get_secret():
    settings = Settings(security__secrets_backend="aws")

    with patch("boto3.Session") as mock_session:
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": '{"k": "v"}'}
        mock_session.return_value.client.return_value = mock_client

        sm = SecretsManager(settings)
        val = sm.get("my/secret")
        assert json.loads(val)["k"] == "v"
        mock_client.get_secret_value.assert_called_once()


def test_vault_secrets_backend_get_secret():
    settings = Settings(security__secrets_backend="vault")

    with patch("hvac.Client") as mock_client_ctor:
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        # Emulate KV v2 response shape
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"k": "v"}}
        }
        mock_client_ctor.return_value = mock_client

        sm = SecretsManager(settings)
        val = sm.get("path/to/secret")
        assert json.loads(val)["k"] == "v"
        mock_client.secrets.kv.v2.read_secret_version.assert_called_once()
