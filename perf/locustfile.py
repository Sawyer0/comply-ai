import os
import json
import random
from pathlib import Path
from typing import List, Dict
from locust import HttpUser, task, between

GOLDEN_PATHS = [
    Path("tests/fixtures/golden_test_cases.json"),
    Path("tests/fixtures/golden_test_cases_comprehensive.json"),
]


def load_golden_cases(limit: int = 100) -> List[Dict]:
    for p in GOLDEN_PATHS:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                cases = data.get("test_cases", [])
                random.shuffle(cases)
                return cases[:limit]
    # Fallback minimal cases if fixtures are absent
    return [
        {"detector": "deberta-toxicity", "input_output": "toxic"},
        {"detector": "regex-pii", "input_output": "email"},
    ]


class MapperUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.api_key_header = os.getenv("MAPPER_API_KEY_HEADER", "X-API-Key")
        self.tenant_header = os.getenv("MAPPER_TENANT_HEADER", "X-Tenant-ID")
        self.api_key = os.getenv("MAPPER_PERF_API_KEY", "")
        self.tenant_id = os.getenv("MAPPER_PERF_TENANT_ID", "perf-tenant")
        self.idempotency_key = os.getenv("MAPPER_PERF_IDEMPOTENCY_KEY", "")
        self.headers = {}
        if self.api_key:
            self.headers[self.api_key_header] = self.api_key
        if self.idempotency_key:
            self.headers["Idempotency-Key"] = self.idempotency_key
        # Note: tenant_id goes in body (MapperAPI syncs header/body if present)
        self.cases = load_golden_cases(limit=int(os.getenv("MAPPER_PERF_CASE_LIMIT", "100")))

    def _pick_case(self) -> Dict:
        return random.choice(self.cases) if self.cases else {"detector": "regex-pii", "input_output": "email"}

    @task(5)
    def map_single(self):
        case = self._pick_case()
        payload = {
            "detector": case["detector"],
            "output": case.get("input_output", case.get("output", "")),
            "tenant_id": self.tenant_id,
        }
        with self.client.post("/map", json=payload, headers=self.headers, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            data = resp.json()
            if not isinstance(data, dict) or "taxonomy" not in data:
                resp.failure("Invalid response schema: missing 'taxonomy'")
            else:
                resp.success()

    @task(1)
    def map_batch(self):
        batch_size = int(os.getenv("MAPPER_PERF_BATCH_SIZE", "5"))
        items = []
        for _ in range(batch_size):
            case = self._pick_case()
            items.append({
                "detector": case["detector"],
                "output": case.get("input_output", case.get("output", "")),
                "tenant_id": self.tenant_id,
            })
        with self.client.post("/map/batch", json={"requests": items}, headers=self.headers, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            data = resp.json()
            if not isinstance(data, dict) or "results" not in data:
                resp.failure("Invalid batch response schema: missing 'results'")
            else:
                resp.success()
