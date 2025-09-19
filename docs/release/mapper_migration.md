# Llama Mapper Request Schema Migration Plan

Status: Announced
Announce date: 2025-09-19
Sunset date for legacy DetectorRequest: 2025-10-31 (00:00:00 GMT)
Removal target: 0.3.0 (no earlier than 2025-11-15)

Summary
- Preferred request schema for /map: MapperPayload (see .kiro/specs/service-contracts.md Section 3)
- Legacy DetectorRequest is deprecated and will be removed after the sunset period
- During the deprecation window, responses include:
  - Header: Deprecation: true
  - Header: Sunset: Fri, 31 Oct 2025 00:00:00 GMT
  - Header: Link: <https://github.com/your-org/comply-ai/blob/main/docs/release/mapper_migration.md>; rel="sunset"
  - Metric: mapper_request_deprecated_total{type="DetectorRequest"}

Timeline
- 2025-09-19: Deprecation announced. Mapper accepts both MapperPayload and DetectorRequest.
- 2025-10-31: Sunset date. Clients should have completed migration to MapperPayload.
- >= 2025-11-15: Removal in version 0.3.0—Mapper will stop accepting DetectorRequest.

Client migration guidance
- New usage (MapperPayload):
  {
    "detector": "orchestrated-multi",
    "output": "toxic|hate|pii_detected",
    "tenant_id": "tenant-123",
    "metadata": {
      "contributing_detectors": ["deberta-toxicity", "openai-moderation"],
      "aggregation_method": "weighted_average",
      "coverage_achieved": 1.0,
      "provenance": [
        {"detector":"deberta-toxicity","confidence":0.93}
      ]
    }
  }

- Legacy usage (DetectorRequest; deprecated):
  {
    "detector": "deberta-toxicity",
    "output": "toxic"
  }

Validation and privacy
- Do not include raw customer content in payloads; send only signals/metadata.
- Payload size limit defaults to 64 KB. Raw-content-like payloads or oversize payloads are rejected with 400 INVALID_REQUEST.

OpenAPI and SDKs
- OpenAPI documents oneOf request bodies for /map and /map/batch.
- SDKs should be updated to emit MapperPayload.

Questions
- Contact the platform team if you need a temporary exception or larger payload bounds for a specific tenant.
