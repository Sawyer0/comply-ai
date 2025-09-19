"""
Unit tests for the in-memory rate limiter token bucket.
"""
import asyncio

import pytest

from src.llama_mapper.api.rate_limit import MemoryRateLimiterBackend


@pytest.mark.asyncio
async def test_token_bucket_basic_allow_and_block():
    backend = MemoryRateLimiterBackend()
    endpoint = "map"
    identity = "id1"
    limit = 2
    window = 60

    r1 = await backend.allow(endpoint, identity, limit, window)
    assert r1.allowed
    assert r1.remaining == 1

    r2 = await backend.allow(endpoint, identity, limit, window)
    assert r2.allowed
    assert r2.remaining == 0

    r3 = await backend.allow(endpoint, identity, limit, window)
    assert not r3.allowed
    assert r3.remaining == 0
    assert r3.reset_seconds > 0
    assert r3.reset_seconds <= window


@pytest.mark.asyncio
async def test_token_bucket_refill_over_time():
    backend = MemoryRateLimiterBackend()
    endpoint = "map"
    identity = "id2"
    limit = 2
    window = 1  # Fast window for test: 2 tokens per second

    # Consume two tokens
    r1 = await backend.allow(endpoint, identity, limit, window)
    r2 = await backend.allow(endpoint, identity, limit, window)
    assert r1.allowed and r2.allowed

    # Next should block
    r3 = await backend.allow(endpoint, identity, limit, window)
    assert not r3.allowed

    # Wait half a second; at 2 tokens/sec, ~1 token refills
    await asyncio.sleep(0.6)
    r4 = await backend.allow(endpoint, identity, limit, window)
    assert r4.allowed
    # Remaining should be >= 0
    assert r4.remaining >= 0
