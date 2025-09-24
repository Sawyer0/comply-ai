# Lightweight stub for clickhouse_driver to allow tests to run without the real package installed.
# This provides a minimal Client interface that can be patched by tests.

from typing import Any


class Client:  # type: ignore
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def execute(self, *args: Any, **kwargs: Any) -> None:
        return None
