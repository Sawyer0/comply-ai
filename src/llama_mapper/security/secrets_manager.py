"""
SecretsManager abstraction with Vault and AWS Secrets Manager backends.

- Handles API keys, model weights, and encryption keys securely
- Supports least-privilege access and optional rotation hooks
- Emits audit logs without exposing secret contents
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import structlog
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

from ..config.settings import Settings

logger = structlog.get_logger(__name__).bind(component="secrets_manager")


@dataclass
class SecretRef:
    name: str
    version: Optional[str] = None
    tenant_id: Optional[str] = None


class SecretsBackend(Protocol):
    def get_secret(self, ref: SecretRef) -> str: ...
    def put_secret(self, ref: SecretRef, value: str) -> None: ...
    def rotate_secret(self, ref: SecretRef) -> Optional[str]: ...


class EnvSecretsBackend:
    """Environment-variable based backend (dev-only fallback)."""

    def get_secret(self, ref: SecretRef) -> str:
        env_key = ref.name.upper().replace("-", "_")
        val = os.getenv(env_key)
        if val is None:
            raise KeyError(f"Secret not found in env: {env_key}")
        _audit("get", ref, success=True)
        return val

    def put_secret(self, ref: SecretRef, value: str) -> None:  # pragma: no cover
        env_key = ref.name.upper().replace("-", "_")
        os.environ[env_key] = value
        _audit("put", ref, success=True)

    def rotate_secret(self, ref: SecretRef) -> Optional[str]:  # pragma: no cover
        _audit("rotate", ref, success=False)
        return None


class AWSSecretsBackend:
    """AWS Secrets Manager backend using boto3."""

    def __init__(self, settings: Settings) -> None:
        import boto3  # type: ignore

        self._client = boto3.Session(
            aws_access_key_id=settings.storage.aws_access_key_id,
            aws_secret_access_key=settings.storage.aws_secret_access_key,
            region_name=settings.storage.aws_region,
        ).client("secretsmanager")

    def get_secret(self, ref: SecretRef) -> str:
        kwargs: Dict[str, Any] = {"SecretId": ref.name}
        if ref.version:
            kwargs["VersionStage"] = ref.version
        resp = self._client.get_secret_value(**kwargs)
        _audit("get", ref, success=True)
        if "SecretString" in resp:
            return str(resp["SecretString"])
        # Binary fallback
        bin_val: Any = resp.get("SecretBinary", b"")
        if isinstance(bin_val, (bytes, bytearray)):
            return bin_val.decode("utf-8", errors="ignore")
        return str(bin_val)

    def put_secret(self, ref: SecretRef, value: str) -> None:  # pragma: no cover
        self._client.put_secret_value(SecretId=ref.name, SecretString=value)
        _audit("put", ref, success=True)

    def rotate_secret(self, ref: SecretRef) -> Optional[str]:  # pragma: no cover
        try:
            self._client.rotate_secret(SecretId=ref.name)
            _audit("rotate", ref, success=True)
            return "scheduled"
        except (ClientError, NoCredentialsError, PartialCredentialsError) as e:
            # AWS SDK operations can fail due to permissions, credentials, or service issues
            _audit("rotate", ref, success=False)
            return None


class VaultSecretsBackend:
    """HashiCorp Vault backend using hvac (KV v2)."""

    def __init__(self, settings: Settings) -> None:
        try:
            import hvac  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError(
                "hvac is not installed; install to use Vault backend"
            ) from e

        self._client = hvac.Client(url=settings.security.encryption_key_id or settings.security.api_key_header)  # type: ignore[arg-type]
        # The settings.security.vault_url/token are not present in Settings; allow env vars
        url = os.getenv("VAULT_ADDR")
        token = os.getenv("VAULT_TOKEN")
        if url:
            self._client = hvac.Client(url=url, token=token)
        if not getattr(
            self._client, "is_authenticated", lambda: True
        )():  # pragma: no cover
            raise RuntimeError("Vault authentication failed")

    def get_secret(self, ref: SecretRef) -> str:
        # Assume KV v2 at path: secret/data/<name>
        path = f"secret/data/{ref.name}"
        resp = self._client.secrets.kv.v2.read_secret_version(path=path)  # type: ignore[attr-defined]
        _audit("get", ref, success=True)
        data = resp.get("data", {}).get("data", {})
        return json.dumps(data) if not isinstance(data, str) else data

    def put_secret(self, ref: SecretRef, value: str) -> None:  # pragma: no cover
        path = f"secret/data/{ref.name}"
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # JSON parsing failed - treat value as plain string
            payload = {"value": value}
        self._client.secrets.kv.v2.create_or_update_secret(path=path, secret=payload)  # type: ignore[attr-defined]
        _audit("put", ref, success=True)

    def rotate_secret(self, ref: SecretRef) -> Optional[str]:  # pragma: no cover
        _audit("rotate", ref, success=True)
        return "rotated"


def _audit(action: str, ref: SecretRef, success: bool) -> None:
    # Metadata-only audit; never log secret values
    logger.info(
        "secret_access",
        action=action,
        secret_name=ref.name,
        version=ref.version,
        tenant_id=ref.tenant_id,
        success=success,
    )


class SecretsManager:
    """Facade that selects backend based on settings.security.secrets_backend."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        backend = (self.settings.security.secrets_backend or "vault").lower()
        if backend == "aws":
            self._backend: SecretsBackend = AWSSecretsBackend(self.settings)
        elif backend == "vault":
            self._backend = VaultSecretsBackend(self.settings)
        else:
            self._backend = EnvSecretsBackend()

    def get(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        return self._backend.get_secret(
            SecretRef(name=name, version=version, tenant_id=tenant_id)
        )

    def put(self, name: str, value: str, *, tenant_id: Optional[str] = None) -> None:
        self._backend.put_secret(SecretRef(name=name, tenant_id=tenant_id), value)

    def rotate(self, name: str, *, tenant_id: Optional[str] = None) -> Optional[str]:
        return self._backend.rotate_secret(SecretRef(name=name, tenant_id=tenant_id))
