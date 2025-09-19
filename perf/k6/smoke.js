import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: Number(__ENV.K6_VUS || 5),
  duration: __ENV.K6_DURATION || "30s",
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<250"], // ms
  },
};

const BASE_URL = __ENV.PERF_BASE_URL || "http://localhost:8000";
const API_KEY_HEADER = __ENV.MAPPER_API_KEY_HEADER || "X-API-Key";
const API_KEY = __ENV.MAPPER_PERF_API_KEY || "";
const TENANT_ID = __ENV.MAPPER_PERF_TENANT_ID || "perf-tenant";

const headers = { "Content-Type": "application/json" };
if (API_KEY) headers[API_KEY_HEADER] = API_KEY;

const cases = [
  { detector: "deberta-toxicity", output: "toxic" },
  { detector: "regex-pii", output: "email" },
  { detector: "llama-guard", output: "violence" },
];

export default function () {
  const pick = cases[Math.floor(Math.random() * cases.length)];
  const payload = JSON.stringify({
    detector: pick.detector,
    output: pick.output,
    tenant_id: TENANT_ID,
  });

  const res = http.post(`${BASE_URL}/map`, payload, { headers });
  const ok = check(res, {
    "status is 200": (r) => r.status === 200,
    "has taxonomy": (r) => {
      try {
        const j = r.json();
        return j && j.taxonomy && Array.isArray(j.taxonomy);
      } catch (_) {
        return false;
      }
    },
  });

  sleep(0.2);
}
