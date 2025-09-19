import json
import os
import random
from pathlib import Path

import pytest
import requests

BASE_URL = os.getenv("PERF_BASE_URL")
API_KEY_HEADER = os.getenv("MAPPER_API_KEY_HEADER", "X-API-Key")
API_KEY = os.getenv("MAPPER_PERF_API_KEY", "")
TENANT_ID = os.getenv("MAPPER_PERF_TENANT_ID", "perf-tenant")

GOLDEN_FILES = [
    Path("tests/fixtures/golden_test_cases.json"),
    Path("tests/fixtures/golden_test_cases_comprehensive.json"),
]

requires_base_url = pytest.mark.skipif(
    not BASE_URL, reason="PERF_BASE_URL not set; skipping HTTP conformance tests"
)


def load_cases(limit: int = 50):
    for p in GOLDEN_FILES:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                cases = data.get("test_cases", [])
                random.shuffle(cases)
                return cases[:limit]
    return []


@requires_base_url
@pytest.mark.parametrize(
    "case", load_cases(limit=int(os.getenv("MAPPER_PERF_CASE_LIMIT", "25")))
)
def test_map_conformance_http(case):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers[API_KEY_HEADER] = API_KEY

    payload = {
        "detector": case["detector"],
        "output": case.get("input_output", case.get("output", "")),
        "tenant_id": TENANT_ID,
    }

    resp = requests.post(f"{BASE_URL}/map", json=payload, headers=headers, timeout=10)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    assert isinstance(data, dict)
    assert "taxonomy" in data and isinstance(data["taxonomy"], list)


@requires_base_url
def test_map_batch_conformance_http():
    cases = load_cases(limit=10)
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers[API_KEY_HEADER] = API_KEY

    items = [
        {
            "detector": c["detector"],
            "output": c.get("input_output", c.get("output", "")),
            "tenant_id": TENANT_ID,
        }
        for c in cases
    ]

    resp = requests.post(
        f"{BASE_URL}/map/batch", json={"requests": items}, headers=headers, timeout=15
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    assert isinstance(data, dict)
    assert "results" in data and isinstance(data["results"], list)
