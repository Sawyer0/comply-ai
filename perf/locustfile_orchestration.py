import os
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner

# Test data paths
GOLDEN_PATHS = [
    Path("tests/fixtures/golden_test_cases.json"),
    Path("tests/fixtures/golden_test_cases_comprehensive.json"),
    Path("../tests/fixtures/golden_test_cases.json"),  # Try parent directory
    Path("../tests/fixtures/golden_test_cases_comprehensive.json"),
]

# Realistic test content for orchestration
ORCHESTRATION_CONTENT = {
    "text": [
        "This document contains sensitive information including email addresses like user@example.com and phone numbers such as +1-555-0123. The quarterly financial report shows revenue growth of 15% with total assets valued at $2.3 million.",
        "URGENT: Security vulnerability detected in authentication system. The current implementation allows SQL injection attacks through the login form. User credentials may be compromised. Immediate action required.",
        "Customer data analysis reveals PII information in logs. Found 1,247 email addresses, 89 phone numbers, and 23 social security numbers in application logs over the past 30 days.",
        "Machine learning model training completed. Model achieved 94.2% accuracy on test dataset with F1 score of 0.91. Ready for production deployment with confidence threshold set to 0.85.",
    ],
    "code": [
        """function authenticate(username, password) {
    const query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'";
    database.execute(query); // SQL Injection vulnerability
}""",
        """import requests

def api_call():
    api_key = "sk-1234567890abcdef"  # Hardcoded API key
    response = requests.get("https://api.third-party.com/data", headers={"Authorization": f"Bearer {api_key}"})""",
        """public class UserService {
    private String apiKey = "pk_live_1234567890"; // Exposed API key

    public void processPayment(String cardNumber) {
        // Process payment without proper validation
    }
}""",
    ],
    "document": [
        """CONFIDENTIAL MEMORANDUM

To: Executive Team
From: Security Department
Date: {date}
Subject: Data Breach Investigation Results

Our investigation has identified a data breach affecting 15,673 customer records. The breach occurred on March 15, 2024, when an attacker exploited a zero-day vulnerability in our authentication system.

Compromised data includes:
- Customer names and addresses
- Email addresses: 15,673 records
- Phone numbers: 12,891 records
- Credit card information: 8,234 records
- Social security numbers: 3,456 records

The attacker gained access through compromised employee credentials and maintained persistent access for 17 days before detection.""",
        """CONTRACT FOR SERVICES

This Service Agreement ("Agreement") is entered into as of {date}, by and between:

TechCorp Solutions Inc.
123 Business Avenue
San Francisco, CA 94105

and

Client Corporation
456 Corporate Blvd
New York, NY 10001

Service Description: Development and implementation of custom software solution for financial data processing, including access to sensitive financial records and customer payment information.

Payment Terms: $250,000 total contract value, payable in three installments of $83,333.33 each.

Confidentiality: Both parties agree to maintain strict confidentiality of all proprietary information, trade secrets, and sensitive data accessed during the course of this engagement.""",
    ]
}

# Orchestration scenarios for different load patterns
ORCHESTRATION_SCENARIOS = {
    "high_frequency": {
        "name": "High Frequency Content Analysis",
        "description": "Frequent content analysis requests typical of real-time scanning",
        "weight": 60,
        "content_types": ["text"],
        "priorities": ["normal", "high"],
        "processing_modes": ["sync"],
    },
    "batch_processing": {
        "name": "Batch Content Processing",
        "description": "Batch processing of multiple documents",
        "weight": 25,
        "content_types": ["text", "document", "code"],
        "priorities": ["normal"],
        "processing_modes": ["sync"],
    },
    "critical_security": {
        "name": "Critical Security Analysis",
        "description": "High-priority security content analysis",
        "weight": 10,
        "content_types": ["text", "code"],
        "priorities": ["high", "critical"],
        "processing_modes": ["sync"],
    },
    "async_processing": {
        "name": "Async Content Processing",
        "description": "Asynchronous processing for large content",
        "weight": 5,
        "content_types": ["document", "code"],
        "priorities": ["normal"],
        "processing_modes": ["async"],
    },
}


def load_golden_cases(limit: int = 200) -> List[Dict]:
    """Load test cases from golden fixtures"""
    for p in GOLDEN_PATHS:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    cases = data.get("test_cases", [])
                    random.shuffle(cases)
                    return cases[:limit]
            except Exception as e:
                print(f"Failed to load {p}: {e}")
                continue

    # Fallback to synthetic cases if fixtures are absent
    return [
        {"detector": "deberta-toxicity", "output": "toxic"},
        {"detector": "regex-pii", "output": "email"},
        {"detector": "llama-guard", "output": "violence"},
    ]


def generate_content(content_type: str) -> str:
    """Generate realistic test content based on type"""
    templates = ORCHESTRATION_CONTENT[content_type]
    return random.choice(templates).format(date=datetime.now().strftime("%B %d, %Y"))


def create_orchestration_request(scenario: str, tenant_id: str) -> Dict:
    """Create a realistic orchestration request"""
    scenario_config = ORCHESTRATION_SCENARIOS[scenario]

    content_type = random.choice(scenario_config["content_types"])
    content = generate_content(content_type)

    return {
        "content": content,
        "content_type": content_type,
        "tenant_id": tenant_id,
        "policy_bundle": "security-scan" if "security" in scenario else "default",
        "environment": random.choice(["dev", "stage", "prod"]),
        "processing_mode": random.choice(scenario_config["processing_modes"]),
        "priority": random.choice(scenario_config["priorities"]),
        "metadata": {
            "scenario": scenario,
            "request_source": "load-test",
            "content_length": len(content),
            "generated_at": datetime.utcnow().isoformat(),
        },
        "required_detectors": ["regex-pii", "deberta-toxicity"] if random.random() > 0.7 else None,
    }


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test with environment setup"""
    print("🚀 Starting Orchestration Load Test")
    print(f"Available scenarios: {list(ORCHESTRATION_SCENARIOS.keys())}")
    if isinstance(environment.runner, MasterRunner):
        print("📊 Running in distributed mode")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Cleanup after test completion"""
    print("✅ Orchestration Load Test completed")


class OrchestrationUser(HttpUser):
    """User class for testing detector orchestration service"""

    # Wait time between requests (realistic think time)
    wait_time = between(0.1, 2.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tenant_id = os.getenv("ORCHESTRATION_PERF_TENANT_ID", "perf-tenant")
        self.api_key_header = os.getenv("ORCHESTRATION_API_KEY_HEADER", "X-API-Key")
        self.api_key = os.getenv("ORCHESTRATION_PERF_API_KEY", "")
        self.policy_bundle = os.getenv("ORCHESTRATION_POLICY_BUNDLE", "default")
        self.idempotency_key = os.getenv("ORCHESTRATION_PERF_IDEMPOTENCY_KEY", "")

        # Headers setup
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers[self.api_key_header] = self.api_key
        if self.idempotency_key:
            self.headers["Idempotency-Key"] = self.idempotency_key

        # Test scenarios with weights
        self.scenarios = list(ORCHESTRATION_SCENARIOS.keys())
        self.scenario_weights = [ORCHESTRATION_SCENARIOS[s]["weight"] for s in self.scenarios]

        # Health check counter
        self.health_check_count = 0

    def on_start(self):
        """Initialize user session"""
        print(f"👤 User started for tenant: {self.tenant_id}")

        # Verify service health
        if not self._verify_service_health():
            print("❌ Service health check failed, user will exit")
            self.environment.runner.quit()

    def _verify_service_health(self) -> bool:
        """Verify orchestration service is healthy"""
        try:
            with self.client.get(
                "/health",
                headers=self.headers,
                catch_response=True,
                timeout=5
            ) as response:
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        print(f"✅ Service healthy: {data.get('detectors_healthy', 0)}/{data.get('detectors_total', 0)} detectors available")
                        return True

                response.failure(f"Health check failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

    def _select_scenario(self) -> str:
        """Select test scenario based on weights"""
        return random.choices(self.scenarios, weights=self.scenario_weights)[0]

    @task(70)
    def orchestrate_single(self):
        """Test single orchestration request"""
        scenario = self._select_scenario()
        payload = create_orchestration_request(scenario, self.tenant_id)

        start_time = time.time()
        with self.client.post(
            "/orchestrate",
            json=payload,
            headers=self.headers,
            catch_response=True,
            timeout=30
        ) as response:
            duration = (time.time() - start_time) * 1000  # Convert to ms

            if response.status_code in [200, 206]:
                # Success or partial success
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        # Validate response structure
                        if "processing_mode" in data and "detector_results" in data:
                            response.success()
                            # Record custom metrics
                            if hasattr(self.environment, 'events'):
                                self.environment.events.request.fire(
                                    request_type="orchestrate",
                                    name="orchestrate_single",
                                    response_time=duration,
                                    response_length=len(response.content) if response.content else 0,
                                    exception=None,
                                )
                        else:
                            response.failure("Invalid response structure")
                    else:
                        response.failure("Response is not a dict")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(20)
    def orchestrate_batch(self):
        """Test batch orchestration requests"""
        batch_size = random.randint(2, 8)
        requests = []

        for _ in range(batch_size):
            scenario = self._select_scenario()
            requests.append(create_orchestration_request(scenario, self.tenant_id))

        payload = {"requests": requests}

        start_time = time.time()
        with self.client.post(
            "/orchestrate/batch",
            json=payload,
            headers=self.headers,
            catch_response=True,
            timeout=60
        ) as response:
            duration = (time.time() - start_time) * 1000

            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and "results" in data:
                        if len(data["results"]) == batch_size:
                            response.success()
                            # Record custom metrics
                            if hasattr(self.environment, 'events'):
                                self.environment.events.request.fire(
                                    request_type="orchestrate",
                                    name=f"orchestrate_batch_{batch_size}",
                                    response_time=duration,
                                    response_length=len(response.content) if response.content else 0,
                                    exception=None,
                                )
                        else:
                            response.failure(f"Expected {batch_size} results, got {len(data.get('results', []))}")
                    else:
                        response.failure("Missing 'results' in batch response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(5)
    def async_orchestrate(self):
        """Test async orchestration processing"""
        scenario = self._select_scenario()
        payload = create_orchestration_request(scenario, self.tenant_id)
        payload["processing_mode"] = "async"

        start_time = time.time()
        with self.client.post(
            "/orchestrate",
            json=payload,
            headers=self.headers,
            catch_response=True,
            timeout=10
        ) as response:
            duration = (time.time() - start_time) * 1000

            if response.status_code == 202:
                try:
                    data = response.json()
                    if isinstance(data, dict) and "job_id" in data:
                        response.success()
                        # Store job_id for potential status checking
                        if not hasattr(self, 'async_jobs'):
                            self.async_jobs = []
                        self.async_jobs.append(data["job_id"])

                        # Record custom metrics
                        if hasattr(self.environment, 'events'):
                            self.environment.events.request.fire(
                                request_type="orchestrate",
                                name="orchestrate_async",
                                response_time=duration,
                                response_length=len(response.content) if response.content else 0,
                                exception=None,
                            )
                    else:
                        response.failure("Missing 'job_id' in async response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 200:
                # Sync processing completed quickly
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(5)
    def health_check(self):
        """Test health check endpoint"""
        self.health_check_count += 1

        with self.client.get(
            "/health",
            headers=self.headers,
            catch_response=True,
            timeout=5
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure(f"Service not healthy: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(2)
    def check_job_status(self):
        """Check status of async jobs (if any)"""
        if not hasattr(self, 'async_jobs') or not self.async_jobs:
            return

        # Check a random job
        job_id = random.choice(self.async_jobs)

        with self.client.get(
            f"/orchestrate/status/{job_id}",
            headers=self.headers,
            catch_response=True,
            timeout=5
        ) as response:
            if response.status_code in [200, 404]:  # 404 means job not found, which is OK for old jobs
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")


class FaultToleranceUser(HttpUser):
    """User class for testing fault tolerance scenarios"""

    wait_time = between(0.5, 3.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tenant_id = os.getenv("ORCHESTRATION_PERF_TENANT_ID", "perf-tenant")
        self.api_key_header = os.getenv("ORCHESTRATION_API_KEY_HEADER", "X-API-Key")
        self.api_key = os.getenv("ORCHESTRATION_PERF_API_KEY", "")

        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers[self.api_key_header] = self.api_key

    @task
    def test_detector_failure(self):
        """Test orchestration with failing detectors"""
        payload = {
            "content": generate_content("text"),
            "content_type": "text",
            "tenant_id": self.tenant_id,
            "policy_bundle": "default",
            "environment": "dev",
            "processing_mode": "sync",
            "priority": "normal",
            "metadata": {
                "test_scenario": "detector_failure",
                "simulate_failures": True
            },
            # Request detectors that might fail
            "required_detectors": ["regex-pii", "nonexistent-detector", "another-fake-detector"]
        }

        with self.client.post(
            "/orchestrate",
            json=payload,
            headers=self.headers,
            catch_response=True,
            timeout=30
        ) as response:
            # Should handle partial failures gracefully
            if response.status_code in [200, 206, 502]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task
    def test_large_content(self):
        """Test with large content to trigger timeouts"""
        large_content = "This is a test document. " * 1000  # ~25KB of content

        payload = {
            "content": large_content,
            "content_type": "document",
            "tenant_id": self.tenant_id,
            "policy_bundle": "default",
            "environment": "dev",
            "processing_mode": "sync",
            "priority": "normal",
            "metadata": {
                "test_scenario": "large_content",
                "content_size": len(large_content)
            }
        }

        with self.client.post(
            "/orchestrate",
            json=payload,
            headers=self.headers,
            catch_response=True,
            timeout=60
        ) as response:
            # Should handle large content or return appropriate error
            if response.status_code in [200, 206, 400, 408]:
                response.success()
            else:
                response.failure(f"Unexpected status for large content: {response.status_code}")

    @task
    def test_malformed_requests(self):
        """Test with malformed requests to verify error handling"""
        test_cases = [
            {"content": "", "content_type": "text"},  # Empty content
            {"content": "test", "content_type": "invalid_type"},  # Invalid content type
            {"content": "test"},  # Missing required fields
            {"tenant_id": self.tenant_id},  # Missing content
        ]

        for i, malformed_payload in enumerate(test_cases):
            with self.client.post(
                "/orchestrate",
                json=malformed_payload,
                headers=self.headers,
                catch_response=True,
                timeout=10
            ) as response:
                # Should return appropriate error codes
                if response.status_code in [400, 422]:
                    response.success()
                else:
                    response.failure(f"Test case {i}: Expected 4xx, got {response.status_code}")


# Configuration for different test scenarios
class OrchestrationLoadTest(HttpUser):
    """Load testing configuration for different scenarios"""

    @task
    def smoke_test(self):
        """Light load test for basic functionality"""
        self.wait_time = between(0.5, 1.0)
        payload = create_orchestration_request("high_frequency", self.tenant_id)
        payload["metadata"]["load_test"] = "smoke"

        with self.client.post("/orchestrate", json=payload, headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Smoke test failed: {response.status_code}")

    @task
    def stress_test(self):
        """Heavy load test with high concurrency"""
        self.wait_time = between(0.1, 0.3)
        # Create multiple concurrent requests
        for _ in range(random.randint(1, 3)):
            payload = create_orchestration_request("high_frequency", self.tenant_id)
            payload["metadata"]["load_test"] = "stress"

            with self.client.post("/orchestrate", json=payload, headers=self.headers, catch_response=True) as response:
                if response.status_code in [200, 206]:
                    response.success()
                else:
                    response.failure(f"Stress test failed: {response.status_code}")


if __name__ == "__main__":
    # Allow running with different user classes
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "fault_tolerance":
            print("🛡️ Running fault tolerance tests")
        elif test_type == "load_test":
            print("⚡ Running load tests")
        elif test_type == "smoke":
            print("🔥 Running smoke tests")
    else:
        print("🚀 Running standard orchestration tests")
