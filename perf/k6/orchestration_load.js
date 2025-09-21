import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";

// Custom metrics
const orchestrateDuration = new Trend("orchestrate_duration");
const orchestrateReqs = new Rate("orchestrate_success_rate");
const batchDuration = new Trend("batch_orchestrate_duration");
const batchReqs = new Rate("batch_orchestrate_success_rate");
const healthCheckReqs = new Rate("health_check_success_rate");

// Configuration from environment
export const options = {
  scenarios: {
    // Smoke test scenario - light load
    smoke: {
      executor: "constant-vus",
      vus: Number(__ENV.K6_SMOKE_VUS || 5),
      duration: __ENV.K6_SMOKE_DURATION || "2m",
      gracefulStop: "30s",
      tags: { test_type: "smoke" },
    },
    // Load test scenario - normal load
    load: {
      executor: "ramping-vus",
      startVUs: Number(__ENV.K6_LOAD_START_VUS || 10),
      stages: [
        { duration: "2m", target: Number(__ENV.K6_LOAD_TARGET_VUS || 50) },
        { duration: "5m", target: Number(__ENV.K6_LOAD_TARGET_VUS || 50) },
        { duration: "2m", target: 0 },
      ],
      gracefulStop: "30s",
      tags: { test_type: "load" },
      startTime: __ENV.K6_SMOKE_DURATION || "2m",
    },
    // Stress test scenario - high load
    stress: {
      executor: "ramping-vus",
      startVUs: Number(__ENV.K6_STRESS_START_VUS || 50),
      stages: [
        { duration: "2m", target: Number(__ENV.K6_STRESS_PEAK_VUS || 200) },
        { duration: "3m", target: Number(__ENV.K6_STRESS_PEAK_VUS || 200) },
        { duration: "2m", target: 0 },
      ],
      gracefulStop: "30s",
      tags: { test_type: "stress" },
      startTime: (__ENV.K6_SMOKE_DURATION || "2m") + "," + (__ENV.K6_LOAD_DURATION || "7m"),
    },
  },
  thresholds: {
    orchestrate_duration: ["p(95)<2000"], // <2s p95 requirement
    orchestrate_success_rate: ["rate>0.95"],
    batch_orchestrate_success_rate: ["rate>0.95"],
    health_check_success_rate: ["rate>0.99"],
    http_req_failed: ["rate<0.05"],
  },
  summaryTrendStats: ["avg", "min", "med", "max", "p(95)", "p(99)"],
};

const BASE_URL = __ENV.PERF_BASE_URL || "http://localhost:8000";
const API_KEY_HEADER = __ENV.ORCHESTRATION_API_KEY_HEADER || "X-API-Key";
const API_KEY = __ENV.ORCHESTRATION_PERF_API_KEY || "";
const TENANT_ID = __ENV.ORCHESTRATION_PERF_TENANT_ID || "perf-tenant";
const TENANT_HEADER = __ENV.ORCHESTRATION_TENANT_HEADER || "X-Tenant-ID";
const POLICY_BUNDLE = __ENV.ORCHESTRATION_POLICY_BUNDLE || "default";

const headers = { "Content-Type": "application/json" };
if (API_KEY) headers[API_KEY_HEADER] = API_KEY;
headers[TENANT_HEADER] = TENANT_ID;

// Test data - realistic content samples for different content types
const testContent = {
  text: [
    "This is a sample text that might contain sensitive information like email addresses such as user@example.com or phone numbers like 555-123-4567.",
    "This document discusses company financial performance and quarterly results. Revenue increased by 15% year-over-year.",
    "The security vulnerability in the authentication system allows unauthorized access to user accounts. This needs immediate attention.",
  ],
  code: [
    "function authenticate(user, password) { if (password === 'admin123') { return true; } else { return false; } }",
    "SELECT * FROM users WHERE email = 'user@domain.com' AND password = '" + Math.random().toString(36) + "';",
    "import requests; response = requests.get('https://api.example.com/data', auth=('user', 'secret_password'))",
  ],
  document: [
    "MEMORANDUM\n\nSubject: Q4 2024 Financial Results\n\nDear Team,\n\nOur quarterly results show significant improvement...",
    "CONTRACT AGREEMENT\n\nThis agreement is made between Party A and Party B for the provision of services...",
    "SECURITY INCIDENT REPORT\n\nIncident ID: SEC-2024-001\nDescription: Unauthorized access detected...",
  ],
};

// Content types for routing
const contentTypes = ["text", "document", "code"];

// Test orchestration request with realistic data
function createOrchestrationRequest(contentType = "text") {
  const content = testContent[contentType][Math.floor(Math.random() * testContent[contentType].length)];

  return {
    content: content,
    content_type: contentType,
    tenant_id: TENANT_ID,
    policy_bundle: POLICY_BUNDLE,
    environment: "dev",
    processing_mode: "sync",
    priority: Math.random() > 0.8 ? "high" : "normal",
    metadata: {
      source: "load-test",
      request_id: `test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    }
  };
}

// Single orchestration test
export function orchestrateTest() {
  const contentType = contentTypes[Math.floor(Math.random() * contentTypes.length)];
  const payload = JSON.stringify(createOrchestrationRequest(contentType));

  const startTime = new Date().getTime();
  const response = http.post(`${BASE_URL}/orchestrate`, payload, {
    headers: headers,
    timeout: "10s"
  });
  const duration = new Date().getTime() - startTime;

  orchestrateDuration.add(duration);

  const success = check(response, {
    "orchestrate status is 200": (r) => r.status === 200,
    "orchestrate status is 206": (r) => r.status === 206, // Partial coverage is acceptable
    "orchestrate has valid response": (r) => {
      try {
        const data = r.json();
        return data && typeof data === "object" && data.processing_mode;
      } catch (e) {
        return false;
      }
    },
    "orchestrate duration < 2s": (r) => duration < 2000,
  });

  orchestrateReqs.add(success);
  sleep(Math.random() * 0.5 + 0.1); // 0.1-0.6s wait time
}

// Batch orchestration test
export function batchOrchestrateTest() {
  const batchSize = Math.floor(Math.random() * 5) + 1; // 1-5 requests per batch
  const requests = [];

  for (let i = 0; i < batchSize; i++) {
    const contentType = contentTypes[Math.floor(Math.random() * contentTypes.length)];
    requests.push(createOrchestrationRequest(contentType));
  }

  const payload = JSON.stringify({ requests: requests });
  const startTime = new Date().getTime();
  const response = http.post(`${BASE_URL}/orchestrate/batch`, payload, {
    headers: headers,
    timeout: "30s"
  });
  const duration = new Date().getTime() - startTime;

  batchDuration.add(duration);

  const success = check(response, {
    "batch orchestrate status is 200": (r) => r.status === 200,
    "batch orchestrate has results": (r) => {
      try {
        const data = r.json();
        return data && Array.isArray(data.results);
      } catch (e) {
        return false;
      }
    },
  });

  batchReqs.add(success);
  sleep(Math.random() * 1 + 0.5); // 0.5-1.5s wait time for batch operations
}

// Health check test
export function healthCheckTest() {
  const response = http.get(`${BASE_URL}/health`, {
    headers: headers,
    timeout: "5s"
  });

  const success = check(response, {
    "health check status is 200": (r) => r.status === 200,
    "health check has status": (r) => {
      try {
        const data = r.json();
        return data && data.status === "healthy";
      } catch (e) {
        return false;
      }
    },
  });

  healthCheckReqs.add(success);
  sleep(0.05); // Very short wait for health checks
}

// Main test function
export default function () {
  const testType = Math.random();

  if (testType < 0.7) {
    // 70% single orchestration requests
    orchestrateTest();
  } else if (testType < 0.9) {
    // 20% batch orchestration requests
    batchOrchestrateTest();
  } else {
    // 10% health checks
    healthCheckTest();
  }
}

// Setup function for test initialization
export function setup() {
  console.log(`Starting orchestration load test against ${BASE_URL}`);
  console.log(`Tenant: ${TENANT_ID}, Policy: ${POLICY_BUNDLE}`);

  // Initial health check to ensure service is ready
  const healthResponse = http.get(`${BASE_URL}/health`, {
    headers: headers,
    timeout: "5s"
  });

  if (healthResponse.status !== 200) {
    throw new Error(`Service health check failed: ${healthResponse.status} ${healthResponse.body}`);
  }

  console.log("Service health check passed");
}

// Teardown function for cleanup
export function teardown(data) {
  console.log("Orchestration load test completed");
}
