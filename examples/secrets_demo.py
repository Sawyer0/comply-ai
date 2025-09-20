#!/usr/bin/env python3
"""
Small demo showing SecretsManager usage and PII redaction.
"""
from __future__ import annotations

import os

from llama_mapper.config.settings import Settings
from llama_mapper.security.redaction import redact_text
from llama_mapper.security.secrets_manager import SecretsManager


def main() -> None:
    # Configure to use env backend for demo
    os.environ.setdefault("MY_API_KEY", "demo-api-key")
    settings = Settings(security__secrets_backend="env")

    sm = SecretsManager(settings)
    api_key = sm.get("MY_API_KEY")
    print("Loaded API key (length only):", len(api_key))

    # Redaction demo
    text = "contact: bob@example.com, card 4111 1111 1111 1111"
    print("Redacted:", redact_text(text))


if __name__ == "__main__":
    main()
