#!/usr/bin/env python3
"""
Generate demo data for Grafana dashboard testing.
Makes requests with different tenants to show per-tenant metrics.
"""

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# Demo configuration
API_BASE = "http://localhost:8001"
TENANTS = [
    "acme-corp",
    "global-bank",
    "healthcare-plus",
    "fintech-startup",
    "enterprise-retail",
]
DEMO_DURATION = 300  # 5 minutes
REQUEST_RATE = 2  # requests per second per tenant


def create_tenant_request(tenant: str, scenario: str = "normal") -> dict:
    """Create a request for a specific tenant with different scenarios."""
    base_request = {
        "request": {
            "period": "2024-01-15T10:30:00Z/2024-01-15T11:30:00Z",
            "tenant": tenant,
            "app": f"{tenant}-app",
            "route": "/api/analysis",
            "required_detectors": ["toxicity", "regex-pii"],
            "observed_coverage": {"toxicity": 0.95, "regex-pii": 0.93},
            "required_coverage": {"toxicity": 0.95, "regex-pii": 0.95},
            "detector_errors": {},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": f"{tenant}-policy-1.0",
            "env": "prod",
        }
    }

    # Modify request based on scenario
    if scenario == "coverage_gap":
        base_request["request"]["observed_coverage"] = {
            "toxicity": 0.80,
            "regex-pii": 0.85,
        }
    elif scenario == "high_fp":
        base_request["request"]["false_positive_bands"] = [
            {
                "detector": "toxicity",
                "score_min": 0.6,
                "score_max": 0.8,
                "fp_rate": 0.40,
            }
        ]
    elif scenario == "incident":
        base_request["request"]["high_sev_hits"] = [
            {
                "detector": "toxicity",
                "taxonomy": "HARM.SPEECH.Toxicity",
                "count": 15,
                "p95_score": 0.95,
            }
        ]
    elif scenario == "error":
        # This will cause a 400 error
        base_request["request"]["env"] = "invalid_env"

    return base_request


def make_request(tenant: str, scenario: str = "normal") -> bool:
    """Make a single request for a tenant."""
    try:
        request_data = create_tenant_request(tenant, scenario)

        headers = {"Content-Type": "application/json", "X-Tenant": tenant}

        response = requests.post(
            f"{API_BASE}/api/v1/analysis/analyze",
            json=request_data,
            headers=headers,
            timeout=10,
        )

        success = response.status_code == 200
        if not success:
            print(f"⚠️  {tenant}: {response.status_code} - {scenario}")
        else:
            result = response.json()
            confidence = result.get("confidence", 0)
            print(f"✅ {tenant}: confidence={confidence:.2f}, scenario={scenario}")

        return success

    except Exception as e:
        print(f"❌ {tenant}: Error - {str(e)}")
        return False


def tenant_worker(tenant: str, stop_event: threading.Event):
    """Worker function for a single tenant."""
    request_count = 0
    success_count = 0

    while not stop_event.is_set():
        # Determine scenario (most requests are normal, some are special)
        scenario_roll = random.random()
        if scenario_roll < 0.7:
            scenario = "normal"
        elif scenario_roll < 0.8:
            scenario = "coverage_gap"
        elif scenario_roll < 0.85:
            scenario = "high_fp"
        elif scenario_roll < 0.9:
            scenario = "incident"
        else:
            scenario = "error"  # Simulate some errors

        success = make_request(tenant, scenario)
        request_count += 1
        if success:
            success_count += 1

        # Wait between requests (with some jitter)
        sleep_time = (1.0 / REQUEST_RATE) + random.uniform(-0.1, 0.1)
        time.sleep(max(0.1, sleep_time))

    success_rate = (success_count / request_count) * 100 if request_count > 0 else 0
    print(f"🏁 {tenant}: {request_count} requests, {success_rate:.1f}% success rate")


def main():
    """Generate demo data for the dashboard."""
    print("🎪 Starting Grafana Dashboard Demo Data Generation")
    print(
        f"📊 Generating traffic for {len(TENANTS)} tenants for {DEMO_DURATION} seconds"
    )
    print(f"⚡ Target rate: {REQUEST_RATE} req/sec per tenant")
    print(f"🎯 API: {API_BASE}")
    print("\nPress Ctrl+C to stop early\n")

    # Check if API is available
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        if response.status_code != 200:
            print(f"❌ API not available at {API_BASE}")
            return
    except Exception as e:
        print(f"❌ Cannot reach API: {e}")
        return

    # Start worker threads for each tenant
    stop_event = threading.Event()
    threads = []

    try:
        for tenant in TENANTS:
            thread = threading.Thread(target=tenant_worker, args=(tenant, stop_event))
            thread.start()
            threads.append(thread)

        # Run for the specified duration
        print(f"🚀 Demo running... (will stop after {DEMO_DURATION} seconds)")
        time.sleep(DEMO_DURATION)

    except KeyboardInterrupt:
        print("\n⏹️  Stopping demo due to user interrupt...")

    finally:
        # Stop all threads
        stop_event.set()

        print("\n⏳ Waiting for threads to finish...")
        for thread in threads:
            thread.join(timeout=5)

        print("✅ Demo data generation complete!")
        print(f"\n📈 Check your Grafana dashboard at: http://localhost:3000")
        print("📊 Check Prometheus metrics at: http://localhost:9090")
        print("🚨 Check AlertManager at: http://localhost:9093")


if __name__ == "__main__":
    main()
