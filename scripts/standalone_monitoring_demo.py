#!/usr/bin/env python3
"""
Standalone monitoring demo that works without Docker.
Shows the key metrics and alerts in action.
"""

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API_BASE = "http://localhost:8001"
TENANTS = ["acme-corp", "global-bank", "healthcare-plus", "fintech-startup"]


def check_metrics_endpoint():
    """Check if metrics endpoint is working."""
    try:
        response = requests.get(f"{API_BASE}/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Metrics endpoint working")

            # Look for our custom metrics
            metrics_text = response.text
            custom_metrics = [
                "analysis_requests_total",
                "analysis_request_duration_seconds",
                "analysis_confidence_score",
                "coverage_gap_rate",
            ]

            found_metrics = []
            for metric in custom_metrics:
                if metric in metrics_text:
                    found_metrics.append(metric)

            print(f"📊 Found custom metrics: {found_metrics}")
            return True
        else:
            print(f"❌ Metrics endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach metrics endpoint: {e}")
        return False


def make_analysis_request(tenant: str, scenario: str = "normal"):
    """Make an analysis request."""
    request_data = {
        "request": {
            "period": "2024-01-15T10:30:00Z/2024-01-15T11:30:00Z",
            "tenant": tenant,
            "app": f"{tenant}-app",
            "route": "/api/analysis",
            "required_detectors": ["toxicity", "regex-pii"],
            "observed_coverage": {
                "toxicity": 0.95 if scenario == "normal" else 0.80,
                "regex-pii": 0.93 if scenario == "normal" else 0.75,
            },
            "required_coverage": {"toxicity": 0.95, "regex-pii": 0.95},
            "detector_errors": (
                {} if scenario != "error" else {"toxicity": {"timeout": 10}}
            ),
            "high_sev_hits": (
                []
                if scenario != "incident"
                else [
                    {
                        "detector": "toxicity",
                        "taxonomy": "HARM.SPEECH.Toxicity",
                        "count": 20,
                        "p95_score": 0.98,
                    }
                ]
            ),
            "false_positive_bands": [],
            "policy_bundle": f"{tenant}-policy-1.0",
            "env": "prod",
        }
    }

    headers = {"Content-Type": "application/json", "X-Tenant": tenant}

    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/v1/analysis/analyze",
            json=request_data,
            headers=headers,
            timeout=10,
        )
        latency = (time.time() - start_time) * 1000  # Convert to ms

        if response.status_code == 200:
            result = response.json()
            confidence = result.get("confidence", 0)
            reason = result.get("reason", "")[:50]
            print(
                f"✅ {tenant}: {latency:.0f}ms, confidence={confidence:.2f}, {reason}..."
            )
            return True, latency
        else:
            print(f"❌ {tenant}: {response.status_code} - {latency:.0f}ms")
            return False, latency

    except Exception as e:
        print(f"❌ {tenant}: Error - {str(e)}")
        return False, 0


def simulate_prometheus_queries():
    """Simulate the key Prometheus queries manually."""
    print("\n📊 SIMULATING GRAFANA DASHBOARD QUERIES")
    print("=" * 60)

    try:
        response = requests.get(f"{API_BASE}/metrics", timeout=5)
        if response.status_code != 200:
            print("❌ Cannot fetch metrics")
            return

        metrics_text = response.text

        # Parse metrics manually (simplified)
        print("\n🔍 Key Metrics Found:")

        # Look for request counts
        request_lines = [
            line
            for line in metrics_text.split("\n")
            if "analysis_requests_total" in line and not line.startswith("#")
        ]
        if request_lines:
            print(f"📈 Request Metrics: {len(request_lines)} series")
            for line in request_lines[:3]:  # Show first 3
                print(f"   {line}")

        # Look for duration metrics
        duration_lines = [
            line
            for line in metrics_text.split("\n")
            if "analysis_request_duration_seconds" in line and not line.startswith("#")
        ]
        if duration_lines:
            print(f"⏱️  Latency Metrics: {len(duration_lines)} series")
            for line in duration_lines[:3]:  # Show first 3
                print(f"   {line}")

        # Look for confidence metrics
        confidence_lines = [
            line
            for line in metrics_text.split("\n")
            if "analysis_confidence_score" in line and not line.startswith("#")
        ]
        if confidence_lines:
            print(f"🎯 Confidence Metrics: {len(confidence_lines)} series")
            for line in confidence_lines[:3]:  # Show first 3
                print(f"   {line}")

    except Exception as e:
        print(f"❌ Error fetching metrics: {e}")


def demo_slo_monitoring():
    """Demonstrate SLO monitoring in action."""
    print("\n🚨 SLO MONITORING DEMO")
    print("=" * 40)

    print("📋 Golden SLO Targets:")
    print("  • P95 Latency: < 500ms")
    print("  • Error Rate: < 1%")
    print("  • Success Rate: > 99%")

    print("\n🎪 Generating multi-tenant traffic...")

    success_count = 0
    total_count = 0
    latencies = []

    # Generate traffic for each tenant
    for i in range(20):  # 20 requests total
        tenant = random.choice(TENANTS)
        scenario = (
            "normal"
            if random.random() < 0.9
            else random.choice(["coverage_gap", "incident"])
        )

        success, latency = make_analysis_request(tenant, scenario)
        total_count += 1
        if success:
            success_count += 1
        if latency > 0:
            latencies.append(latency)

        time.sleep(0.5)  # Brief pause between requests

    # Calculate SLO metrics
    if latencies:
        latencies.sort()
        p95_index = int(0.95 * len(latencies))
        p95_latency = (
            latencies[p95_index] if p95_index < len(latencies) else latencies[-1]
        )
        avg_latency = sum(latencies) / len(latencies)
    else:
        p95_latency = 0
        avg_latency = 0

    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    error_rate = (
        ((total_count - success_count) / total_count) * 100 if total_count > 0 else 0
    )

    print(f"\n📊 SLO RESULTS:")
    print(
        f"  📈 Success Rate: {success_rate:.1f}% ({'✅' if success_rate > 99 else '❌'} SLO)"
    )
    print(
        f"  📉 Error Rate: {error_rate:.1f}% ({'✅' if error_rate < 1 else '❌'} SLO)"
    )
    print(f"  ⏱️  Average Latency: {avg_latency:.0f}ms")
    print(
        f"  🚀 P95 Latency: {p95_latency:.0f}ms ({'✅' if p95_latency < 500 else '❌'} SLO)"
    )

    # Simulate alerts
    print(f"\n🚨 ALERT STATUS:")
    if p95_latency > 500:
        print("  🔴 CRITICAL: P95 latency exceeds 500ms SLO")
    if error_rate > 1:
        print("  🔴 CRITICAL: Error rate exceeds 1% SLO")
    if success_rate < 99:
        print("  🟡 WARNING: Success rate below 99% SLO")

    if p95_latency <= 500 and error_rate <= 1 and success_rate >= 99:
        print("  ✅ All SLOs within targets")


def main():
    """Run the standalone monitoring demo."""
    print("🎪 STANDALONE MONITORING DEMO")
    print("=" * 50)
    print("Demonstrates enterprise-grade monitoring without Docker")
    print()

    # Step 1: Check metrics endpoint
    if not check_metrics_endpoint():
        print("\n❌ Metrics endpoint not available. Make sure analysis API is running.")
        return

    # Step 2: Show current metrics
    simulate_prometheus_queries()

    # Step 3: Generate traffic and show SLO monitoring
    demo_slo_monitoring()

    # Step 4: Show final metrics
    print("\n🔄 Final metrics check...")
    simulate_prometheus_queries()

    print("\n🎯 DEMO COMPLETE!")
    print("\n💡 Key Takeaways:")
    print("  ✅ Metrics endpoint exposes Prometheus-compatible metrics")
    print("  ✅ Multi-tenant request tracking with per-tenant SLAs")
    print("  ✅ Real-time SLO monitoring (latency, error rate, success rate)")
    print("  ✅ Golden alerts for P95 > 500ms and error rate > 1%")
    print("  ✅ Enterprise-ready observability stack")

    print(f"\n📊 Access live metrics: {API_BASE}/metrics")
    print("🚀 Ready for Grafana integration when Docker is available!")


if __name__ == "__main__":
    main()
