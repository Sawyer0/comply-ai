from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


class _DummyResp:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {"output": "ok", "confidence": 0.9}

    def json(self):
        return self._body


class _DummyAsyncClient:
    def __init__(self, seq):
        self._seq = seq

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, endpoint, json):
        # Pop next behavior
        behavior = self._seq.pop(0)
        if behavior == "timeout":
            import httpx

            raise httpx.ReadTimeout("timeout")
        if isinstance(behavior, int):
            return _DummyResp(behavior)
        return _DummyResp(200)


@pytest.mark.asyncio
async def test_detector_client_retries_and_recovers(monkeypatch):
    from detector_orchestration.clients import DetectorClient

    # Simulate timeout then 200 OK
    seq = ["timeout", 200]

    class _Factory:
        def __call__(self, timeout):
            return _DummyAsyncClient(seq)

    import detector_orchestration.clients as clients

    monkeypatch.setattr(clients.httpx, "AsyncClient", lambda timeout: _DummyAsyncClient(seq))

    c = DetectorClient(name="ext", endpoint="http://detector/analyze", timeout_ms=100, max_retries=1, auth={})
    res = await c.analyze("content")
    assert res.status.value == "success"


@pytest.mark.asyncio
async def test_detector_client_gives_up_after_retries(monkeypatch):
    from detector_orchestration.clients import DetectorClient
    import detector_orchestration.clients as clients
    # Two timeouts and retries=1 → timeout error
    seq = ["timeout", "timeout"]
    monkeypatch.setattr(clients.httpx, "AsyncClient", lambda timeout: _DummyAsyncClient(seq))

    c = DetectorClient(name="ext", endpoint="http://detector/analyze", timeout_ms=100, max_retries=1, auth={})
    res = await c.analyze("content")
    assert res.status.value in ("timeout", "failed")

